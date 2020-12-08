from src import Parameters
from src import Preprocessing
from src import TextClassifier
from src import Run
		

class Controller(Parameters):
	
	def __init__(self):
		# Preprocessing pipeline
		self.data = self.prepare_data(Parameters.num_words, Parameters.seq_len)
		
		# Initialize the model
		self.model = TextClassifier(Parameters)
		
		# Training - Evaluation pipeline
		Run().train(self.model, self.data, Parameters)
		
		
	@staticmethod
	def prepare_data(num_words, seq_len, source=None):
		# Preprocessing pipeline
		pr = Preprocessing(num_words, seq_len)
		pr.load_data(source=source)
		pr.clean_text()
		pr.text_tokenization()
		pr.build_vocabulary()
		pr.word_to_idx()
		pr.padding_sentences()
		pr.split_data(dontsplit=False if source is None else True)

		return {'x_train': pr.x_train, 'y_train': pr.y_train, 'x_test': pr.x_test, 'y_test': pr.y_test}
	
	def execute(self, inputdata : list):
		pdata = self.prepare_data(Parameters.num_words, Parameters.seq_len, source=['']+inputdata)
		return Run().execute(self.model, pdata)[1:]
		
if __name__ == '__main__':
	controller = Controller()