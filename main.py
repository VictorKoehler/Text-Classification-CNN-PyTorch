from src import Parameters
from src import Preprocessing
from src import TextClassifier
from src import Run
		

class Controller(Parameters):
	
	def __init__(self):
		# Preprocessing pipeline
		self.pr = Preprocessing(Parameters.num_words, Parameters.seq_len)
		self.data = self.prepare_data()
		
		# Initialize the model
		self.model = TextClassifier(Parameters)
		
		# Training - Evaluation pipeline
		Run().train(self.model, self.data, Parameters)
		

	def prepare_data(self, source=None, split=True):
		# Preprocessing pipeline
		pr = self.pr
		pr.load_data(source=source)
		pr.clean_text()
		pr.text_tokenization()
		pr.build_vocabulary()
		pr.word_to_idx()
		pr.padding_sentences()
		if split:
			pr.split_data()
		else:
			pr.commit_data()

		return {'x_train': pr.x_train, 'y_train': pr.y_train, 'x_test': pr.x_test, 'y_test': pr.y_test}
	
	def execute(self, inputdata : list):
		pdata = self.prepare_data(source=['']+inputdata, split=False)
		return Run().execute(self.model, pdata)[1:]

if __name__ == '__main__':
	controller = Controller()