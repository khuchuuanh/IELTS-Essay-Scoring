from lib import *

data_path = 'D:/My Projects/IELTS_Scoring/'
name_df = 'data_split'
data_frame = pd.read_csv(data_path + name_df + '.csv')

class CustomerDataset(pl.LightningDataModule):
    def __init__(self, model_name, data_frame, text_field : list, label_field = str, max_len= 512,batch_size = 32, remove_special_characters = False, **kwargs):
        super().__init__()

        self.model_name = model_name
        self.data_frame = data_frame
        self.text_field = text_field
        self.label_field = label_field
        self.max_len = max_len
        self.batch_size = batch_size
        self.remove_special_characters = remove_special_characters
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        label_encoder = LabelEncoder()
        self.label_encoder = label_encoder.fit(list(self.data_frame[self.label_field]))
    
    def setup(self, stage = None):
        if len(self.text_field) >1:
            self.data_frame['input'] = list(map(self.preprocess_text, self.data_frame[self.text_field[0]] + self.data_frame[self.text_field[1]]))
        else:
            self.data_frame['input'] = list(map(self.preprocess_text, self.data_frame[self.text_field[0]]))
        self.data_frame['label'] = self.data_frame[self.label_field]
        self.train_dataset = self.convert_to_features(self.data_frame[self.data_frame.Data_type == 'train'])
        self.val_dataset = self.convert_to_features(self.data_frame[self.data_frame.Data_type == 'valid'])
        self.test_dataset = self.convert_to_features(self.data_frame[self.data_frame.Data_type == 'test'])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            sampler = RandomSampler(self.train_dataloader),
            num_workers= 2
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size= self.batch_size,
            sampler= RandomSampler(self.val_dataset),
            num_workers= 2
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size= self.batch_size,
            sampler = RandomSampler(self.test_dataset),
            num_workers= 2
        )
    
    def preprocess_text(self, text) -> str:
        replace_chars = list(string.punctuation.replace("'", ""))
        if self.remove_special_characters:
            for char in replace_chars:
                text = text.replace(char, " ")
        else:
            for char in replace_chars:
                text = text.replace(char, " " + char + " ")
        return " ".join(text.lower().split())

    def convert_to_features(self,df) -> TensorDataset:
        sentences = df.input.values
        labels = df.label.values

        features = [[int(i) for i in features.split()] for feature in data_frame.Feature.values]
        features = torch.tensor(features)

        encoder = self.tokenizer.batch_encode_plus(
            sentences.tolist(),
            add_special_tokens = True,
            max_length = self.max_len,
            padding = 'max_length',
            truncation = True,
            return_attention_mask  = True,
            return_tensor = 'pt' 
        )

        input_ids = encoder['input_ids']
        attention_mask = encoder['attention_mask']

        labels =torch.tensor(self.label_encoder.transform(list(labels)))
        return TensorDataset(input_ids, attention_mask, features, labels)

if __name__ == "__main__":
    model_name = 'distilbert-base-uncased'
    text_field = ['Text']
    label_field = 'Grammar'
    data_module = CustomerDataset(model_name, data_frame, text_field, label_field,max_len = 512, batch_size=32, remove_special_characters= True)
    data_module.setup('fit')
    test = next(iter(data_module.train_dataloader()))
    test[0], test[1], test[2], test[3]
