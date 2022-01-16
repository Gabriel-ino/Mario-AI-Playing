from game_setup import App
from os import system


def main():
    app = App()
    app.run_game()


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(err)
        system('pip install -r requirements.txt')
        main()




