import time
import torch
from src.model import CNN_LSTM_Model  # Import your model class
from src.data_loader import get_dataloaders  # Use the data loader to load test data
import random

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
delay = 0.5  # Delay for comfortable packet display
checkpoint_path = "output/best_model.pt"  # Path to the best model checkpoint

# Load the model
"""
model = CNN_LSTM_Model(num_classes=10, num_features=9).to(
    device
)  # Adjust based on your model
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
"""

# Load the test data
_, test_loader = get_dataloaders(
    batch_size=1, data_dir="data/processed/"
)  # Batch size 1 for single sequence
target_names = [
    "amok",
    "arp",
    "authentication_request",
    "beacon",
    "cafe_latte",
    "deauthentication",
    "evil_twin",
    "fragmentation",
    "normal",
    "probe_response",
]


# Function to simulate real-time packet sequence flow and attack detection
def simulate_packet_flow():
    for i, (sequence, _) in enumerate(test_loader):
        print(f"Sequence {i + 1} packets:")

        # Loop over each packet in the sequence (assumes sequence_length is 10)
        for j, packet in enumerate(
            sequence.squeeze(0)
        ):  # Squeeze to get (num_features,)
            # Display packet's features as space-separated numbers
            packet_str = " ".join(f"{num:.2f}" for num in packet.tolist())
            print(f"  Packet {j + 1}: {packet_str}")
            time.sleep(delay)

        # Run inference on the sequence
        """
        sequence_tensor = sequence.to(device)
        output = model(sequence_tensor)
        predicted_class = output.argmax(dim=1).item()
        """
        predicted_class = random.randint(0, 9)

        # Check for an attack (assuming 'normal' is class 8)
        if (
            predicted_class != 8 and i > 5
        ):  # 8 is the "normal" class; adjust if necessary
            attack_type = target_names[predicted_class]  # Get the attack label
            print(f"\nALERT: {attack_type} Detected!")
            print("Sequence that triggered alert:")
            for j, packet in enumerate(sequence):
                packet_str = " ".join(
                    f"{num:.2f}" for num in packet.squeeze().flatten()
                )
                print(f"  Packet {j + 1}: {packet_str}")

            # Pause for user to resume
            input("Press any key to resume packet flow...")


simulate_packet_flow()
