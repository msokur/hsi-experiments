import argparse
import os

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--arg', 
                    help='sum the integers (default: find the max)')
    
    args = parser.parse_args()
    
    print(args)
    
    
    