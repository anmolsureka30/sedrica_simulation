
F1TENTH Simulation Assignment – Sedrica Tech Team

Setup Instructions
	1.	Create a virtual environment

python3 -m venv .venv
source .venv/bin/activate   # On Mac/Linux
.venv\Scripts\activate      # On Windows


	2.	Install dependencies

pip install -r requirements.txt


Running the Tasks

Task 1 – Keyboard Navigation

Run the following from the examples folder:

python examples/drive.py

or

python examples/key.py

Task 2 – LiDAR Simulation

Run the following:

python examples/lid.py


⸻

Directory Structure (important files)

sim_ass/
├── examples/        # Example scripts
│   ├── drive.py     # Keyboard navigation
│   ├── key.py       # Alternative keyboard navigation
│   ├── lid.py       # LiDAR simulation
│   └── ...
├── gym/             # F110 gym environment
├── requirements.txt # Dependencies
└── setup.py


