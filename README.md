# SPACIOUS-NoiseProcessing

**SPACIOUS-NoiseProcessing** is a Python-based toolkit developed by the PIFSC Protected Species Division to support the processing and analysis of underwater noise data collected under the SPACIOUS program. These tools contribute to evaluating noise conditions in marine environments and assessing potential impacts on protected species, including cetaceans, turtles, and seals.

The repository includes core modules for reading and processing acoustic recordings, as well as example scripts demonstrating practical workflows.

<img width="999" height="787" alt="image" src="https://github.com/user-attachments/assets/3af06d4d-5992-467a-998f-aedf73fef275" />

---

## Features

- Modular functions for filtering and processing underwater acoustic data.  
- Example workflow (`Example.py`) illustrating how to use the processing modules.  
- Straightforward structure for users who wish to extend methods or integrate them into larger pipelines.

---

## Getting Started

### Prerequisites

- Python 3.x  
- Common scientific Python packages (e.g., `numpy`, `scipy`).  

A `requirements.txt` file should be added once dependencies are finalized.

### Installation

```bash
git clone https://github.com/PIFSC-Protected-Species-Division/SPACIOUS-NoiseProcessing.git
cd SPACIOUS-NoiseProcessing

# Optional: set up a virtual environment
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies once requirements.txt is added
pip install -r requirements.txt
