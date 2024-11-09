import numpy as np
import torch
from world import World
from agents import Car, RingBuilding, CircleBuilding, Painting
from geometry import Point
import time
from example_dql_agent import DQLAgent  # Assuming DQLAgent is in a separate file named example_dql_agent.py

# Hyperparameters and environment setup
dt = 0.1  # Time steps in seconds
world_width, world_height = 120, 120
inner_building_radius = 30
num_lanes = 2
lane_marker_width = 0.5
num_of_lane_markers = 50
lane_width = 3.5

# Initialize world
w = World(dt, width=world_width, height=world_height, ppm=6)

# Initialize agent with updated parameters
state_size = 5  # Now includes angular position as well
action_size = 15  # 5 steering * 3 throttle combinations
agent = DQLAgent(state_size, action_size, epsilon=0.5, tau=0.01)  # Lower initial epsilon

# Define the function to update car positions in a circular path
def update_car_position(car, center_x, center_y, radius, speed, dt):
    """Update the car's position to move in a circle."""
    # Get current angle based on position relative to the center
    current_angle = np.arctan2(car.center.y - center_y, car.center.x - center_x)
    
    # Update angle based on speed and radius
    angular_velocity = speed / radius  # v = ω * r
    new_angle = current_angle + angular_velocity * dt
    
    # Calculate new position
    new_x = center_x + radius * np.cos(new_angle)
    new_y = center_y + radius * np.sin(new_angle)
    
    # Update car position and heading
    car.center = Point(new_x, new_y)
    car.heading = new_angle + np.pi / 2  # Make car tangent to the circle

# Setup environment
def environment_setup(w):
    cb = CircleBuilding(Point(world_width/2, world_height/2), inner_building_radius, 'gray80')
    w.add(cb)
    rb = RingBuilding(Point(world_width/2, world_height/2), 
                      inner_building_radius + num_lanes * lane_width + (num_lanes - 1) * lane_marker_width,
                      1 + np.sqrt((world_width/2)**2 + (world_height/2)**2), 'gray80')
    w.add(rb)
    
    for lane_no in range(num_lanes - 1):
        lane_markers_radius = inner_building_radius + (lane_no + 1) * lane_width + (lane_no + 0.5) * lane_marker_width
        lane_marker_height = np.sqrt(2 * (lane_markers_radius**2) * (1 - np.cos((2 * np.pi) / (2 * num_of_lane_markers))))
        for theta in np.arange(0, 2 * np.pi, 2 * np.pi / num_of_lane_markers):
            dx = lane_markers_radius * np.cos(theta)
            dy = lane_markers_radius * np.sin(theta)
            w.add(Painting(Point(world_width/2 + dx, world_height/2 + dy), 
                           Point(lane_marker_width, lane_marker_height), 'white', heading=theta))
    
    return w, cb

w, cb = environment_setup(w)

# Initialize cars
c1 = Car(Point(96, 60), np.pi/2)  # Main car controlled by the agent
c1.max_speed = 30.0
w.add(c1)

# Other cars as obstacles
c2 = Car(Point(world_width/2, 95), np.pi, 'yellow')
c3 = Car(Point(world_width/2, 24), 0, 'blue')
c4 = Car(Point(28, 60), np.pi/2, 'green')
for car in [c2, c3, c4]:
    car.max_speed = 30.0
    w.add(car)

# Main training/testing loop with updated reward structure
episodes = 1000
batch_size = 32
for episode in range(episodes):
    # Reset environment and agent
    c1.center = Point(96, 60)
    c1.heading = np.pi/2
    c1.velocity = Point(0, 3.0)
    crossed_start = False
    lap_count = 0
    total_reward = 0
    last_angle = np.arctan2(c1.center.y - world_height/2, c1.center.x - world_width/2)

    # Run episode
    for time_step in range(1000):  # Arbitrary step limit
        # Get the current state with angular position
        state = agent.get_state(c1, cb)
        
        # Choose action
        steering, throttle = agent.act(state)
        c1.set_control(steering, throttle)
        
        # Update other cars' positions
        radius = inner_building_radius + lane_width / 2
        update_car_position(c2, world_width / 2, world_height / 2, radius, 3.0, dt)
        update_car_position(c3, world_width / 2, world_height / 2, radius + lane_width, 3.0, dt)
        update_car_position(c4, world_width / 2, world_height / 2, radius + lane_width, 3.0, dt)

        # Advance the world
        w.tick()
        w.render()
        time.sleep(dt)

        # Calculate reward and check if the episode is done
        distance_to_center = c1.distanceTo(cb)
        desired_distance = inner_building_radius + lane_width / 2
        distance_error = abs(distance_to_center - desired_distance)

        # Reward for staying on track, moving forward, and smoother controls
        reward = 1.0 - 0.1 * distance_error - 0.05 * abs(steering)
        
        # Additional reward for forward progress and completed laps
        current_angle = np.arctan2(c1.center.y - world_height/2, c1.center.x - world_width/2)
        angle_progress = current_angle - last_angle
        if angle_progress > 0:
            reward += 0.5 * angle_progress
        else:
            reward -= 0.5 * abs(angle_progress)

        # Check if car has completed a lap and reward
        if crossed_start and abs(c1.center.x - 96) < lane_width and abs(c1.center.y - 60) < lane_width:
            lap_count += 1
            reward += 100  # Large reward for completing a lap
            crossed_start = False
        
        # Penalize if episode ends due to collision
        if w.collision_exists():
            reward = -1000
            done = True
        else:
            done = False

        total_reward += reward

        # Store experience and train the agent
        next_state = agent.get_state(c1, cb)
        action_index = agent.steering_actions.index(steering) * len(agent.throttle_actions) + agent.throttle_actions.index(throttle)
        agent.remember(state, action_index, reward, next_state, done)
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        agent.update_target_model()

        if done:
            break

    print(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward}, Laps: {lap_count}")
    
    # Save model periodically
    if (episode + 1) % 100 == 0:
        torch.save(agent.model.state_dict(), f"dql_agent_episode_{episode+1}.pth")

w.close()
