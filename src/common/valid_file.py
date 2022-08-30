
class FileNotExistException(Exception):
	def __init__(self,file_path,message="This file does not exists"):
		self.file_path = file_path
		self.message = message
		super().__init__(self.message)
	def __str__(self):
		return f'{self.file_path} -> {self.message}'