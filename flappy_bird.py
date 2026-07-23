import sys
import os

def main():
    print("====================================================")
    print("       Flappy Bird Reinforcement Learning           ")
    print("====================================================")
    print("1. Play Manually (Use SPACEBAR to flap)")
    print("2. Train the DQN Agent")
    print("3. Watch the Trained Agent Play")
    print("====================================================")
    
    try:
        choice = input("Select an option (1-3): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        return

    if choice == '1':
        print("\nStarting manual play mode...")
        run_manual_play()
    elif choice == '2':
        print("\nStarting agent training mode...")
        try:
            render_choice = input("Do you want to render the training? (y/n) [n]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return
        should_render = render_choice == 'y'
        from train import train
        train(render=should_render)
    elif choice == '3':
        print("\nStarting agent play mode...")
        from play import play
        play()
    else:
        print("Invalid choice. Exiting.")

def run_manual_play():
    import gymnasium as gym
    import flappy_bird_gymnasium
    import pygame

    # Initialize environment in human render mode
    env = gym.make("FlappyBird-v0", render_mode="human")
    state, info = env.reset()
    done = False

    pygame.init()
    print("Game started! Use the SPACEBAR to flap. Close the window or press ESC to quit.")

    clock = pygame.time.Clock()

    while not done:
        action = 0  # default: 0 is idle, 1 is flap

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1  # flap
                elif event.key == pygame.K_ESCAPE:
                    done = True

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Limit framerate for manual playability
        clock.tick(30)

    env.close()
    pygame.quit()
    print(f"\nGame Over! Final Score: {info.get('score', 0)}")

if __name__ == "__main__":
    main()