import sys
import yaml
import os
from crew import GameBuilderCrew


def run():
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    print("## Welcome to the Game Crew")
    print('-------------------------------')

    try:
        with open(os.path.join(os.path.dirname(__file__), 'config', 'gamedesign.yaml'), 'r', encoding='utf-8') as file:
            examples = yaml.safe_load(file)
            if not examples:
                raise ValueError("O arquivo YAML está vazio ou mal formatado.")
    except FileNotFoundError:
        raise FileNotFoundError("Arquivo YAML não encontrado. Verifique o caminho.")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Erro ao carregar o arquivo YAML: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while loading the YAML file: {e}")

    inputs = {
        'game' :  examples['example3_snake']
    }
    try:
        game= GameBuilderCrew().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the game: {e}")

    print("\n\n########################")
    print("## Here is the result")
    print("########################\n")
    print("final code for the game:")
    print(game)
    

def train():
    """
    Train the crew for a given number of iterations.
    """

    try:
        with open(os.path.join(os.path.dirname(__file__), 'config', 'gamedesign.yaml'), 'r', encoding='utf-8') as file:
            examples = yaml.safe_load(file)
            if not examples:
                raise ValueError("O arquivo YAML está vazio ou mal formatado.")
    except FileNotFoundError:
        raise FileNotFoundError("Arquivo YAML não encontrado. Verifique o caminho.")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Erro ao carregar o arquivo YAML: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while loading the YAML file: {e}")

    inputs = {
        'game' : examples['example1_pacman']
    }
    try:
        GameBuilderCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


if __name__ == "__main__":
    run()
