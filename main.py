import yaml
# Create a YAML instance


# Specify the path to your YAML file
yaml_file_path = 'imu_data.yaml'  # Replace with the actual path to your YAML file

# Load YAML data from file into a Python dictionary
stream = open(yaml_file_path)
dict = list(yaml.load_all(stream=stream, Loader=yaml.FullLoader))

# Print or use the Python dictionary
for x in range(5):
    print(dict[x])
