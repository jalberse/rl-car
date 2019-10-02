import json
import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':
    filename = sys.argv[1]
    try:
        with open (filename, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(e)
        exit()
    
    # Graph rewards
    # TODO Ensure labels, slicing, etc fine for each generated graph. 
    # Easier to manually change this utility script than make it robust during development.
    plt.plot(data['rewards'][0:6000:1])
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.title('Training run 3')
    plt.show()