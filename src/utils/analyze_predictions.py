import json
import matplotlib.pyplot as plt
from collections import Counter
import os
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_class_distribution(data):
    # Filter out 'full_image' class
    class_counts = Counter(item['class_name'] for item in data if item['class_name'] != 'full_image')
    return class_counts

def plot_distribution(class_counts):
    # Sort by count for better visualization
    classes, counts = zip(*sorted(class_counts.items(), key=lambda x: x[1], reverse=True))
    
    plt.figure(figsize=(12, 8))
    plt.bar(classes, counts)
    plt.xlabel('Class Name', fontsize=12, fontweight='bold')  # Increased font size and weight
    plt.ylabel('Count', fontsize=12, fontweight='bold')  # Increased font size and weight
    plt.title('Class Distribution', fontsize=14, fontweight='bold')  # Increased font size and weight
    plt.xticks(rotation=90, fontsize=10)  # Increased font size for x-axis labels
    plt.yticks(fontsize=10)  # Increased font size for y-axis labels
    plt.subplots_adjust(bottom=0.4)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def print_statistics(class_counts):
    total = sum(class_counts.values())
    print("\nClass Statistics:")
    print(f"Total predictions: {total}")
    print(f"Number of unique classes: {len(class_counts)}")
    print("\nClass Counts:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{class_name}: {count} ({count/total:.1%})")

def ensure_directories_exist():
    """Create necessary directories if they don't exist"""
    os.makedirs('logs/detection_analysis/images', exist_ok=True)
    os.makedirs('logs/detection_analysis/text', exist_ok=True)

def save_plot(class_counts):
    """Save the plot to the images directory"""
    # Sort by count for better visualization
    classes, counts = zip(*sorted(class_counts.items(), key=lambda x: x[1], reverse=True))
    
    plt.figure(figsize=(12, 8))
    plt.bar(classes, counts)
    plt.xlabel('Class Name', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.title('Class Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.subplots_adjust(bottom=0.4)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_path = f'logs/detection_analysis/images/detected_class_distribution.png'
    plt.savefig(image_path)
    plt.close()
    print(f"Saved plot to: {image_path}")

def save_statistics(class_counts, file_path):
    """Save statistics to a text file"""
    total = sum(class_counts.values())
    with open(file_path, 'w') as f:
        f.write("Class Statistics:\n")
        f.write(f"Total predictions: {total}\n")
        f.write(f"Number of unique classes: {len(class_counts)}\n\n")
        f.write("Class Counts:\n")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{class_name}: {count} ({count/total:.1%})\n")
    print(f"Saved statistics to: {file_path}")

def main():
    json_file = 'data/index/product_index.json'
    
    # Ensure directories exist
    ensure_directories_exist()
    
    # Load and analyze data
    data = load_json(json_file)
    class_counts = get_class_distribution(data)
    
    # Generate timestamp for log files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save statistics
    stats_file = f'logs/detection_analysis/text/class_statistics.txt'
    save_statistics(class_counts, stats_file)
    
    # Save plot
    save_plot(class_counts)

if __name__ == '__main__':
    main() 
    