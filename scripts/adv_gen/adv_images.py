

class AdversarialImages():
    def __init__(self, config):
        self.n_estimators = config['n_estimators']
        self.raw_data_path = config['raw_data_path']
        self.main()
    def main(self):
