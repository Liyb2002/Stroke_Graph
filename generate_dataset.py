
import Preprocessing.generate_dataset

def run():
    d_generator = Preprocessing.generate_dataset.dataset_generator()
    d_generator.generate_dataset()

run()