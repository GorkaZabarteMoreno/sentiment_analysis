# Import libraries
library(caret); library(quanteda); library(quanteda.textmodels); library(sentencepiece); library(spacyr); library(stopwords); library(tm); library(tokenizers.bpe); library(utf8);

# Load dataset and check for missing values
dataframe <- read.csv(file = '../data/financial_news.csv', header = FALSE, encoding = "UTF-8");
which(is.na.data.frame(dataframe));

# Split dataframe columns
target_column <- dataframe$V1; 
text_column <- dataframe$V2;

# Text column saves in a unique string to process easier
text <- paste(text_column, collapse = " ");

# UTF-8 checks
which(!utf8_valid(text));
text_normalize <- utf8_normalize(text);
sum(text_normalize != text);

# Target column frequency
target_frequency <- data.frame(table(target_column));
plot(target_frequency[order(target_frequency$Freq, decreasing = TRUE), ]);

# Remove useless words
remove_words <- c(stopwords("en"), 'the', 'eur', 's', 'mn', 'n', 'm');
text <- removeWords(text, remove_words);

# Data cleaning
clean_text <- function(txt){
  txt <- tolower(txt);  
  txt <- gsub("\\d+", "", txt); 
  txt <- gsub("[\x21-\x2F]", "", txt);
  txt <- gsub("[\x3A-\x40]", "", txt); 
  txt <- gsub("[\x5B-\x60]", "", txt);
  txt <- gsub("[\x7B-\x7F]", "", txt);  
  txt <- gsub("\n{1,}", " ", txt);
  txt <- gsub("[ ]{2,}", " ", txt);
}

text_column <- clean_text(text_column); 
text <- clean_text(text);

counter <- 1 
for (i in text_column) {
  text_column[counter] <- removeWords(i, remove_words);
  counter <- counter + 1;
}

# Part of speech with spacyr library
Sys.setenv(RETICULATE_PYTHON = "~/.virtualenvs/spacy_virtualenv/bin/python");

spacy_initialize(model = "en_core_web_sm");

tic <- Sys.time(); word_info <- spacy_parse(unique(text), entity = FALSE, lemma = FALSE); Sys.time()-tic;
tokens <- word_info$token;
plot(head(sort(table(tokens), decreasing = TRUE), n = 10), xlab = "Token", ylab = "Ocurrences");
head(sort(table(tokens), decreasing = TRUE), n = 10)

# Split dataset for byte pair encoding  
split_data <- function(data) {
  data_length <- length(data);
  train_size <- 0.7 * data_length; 
  train_data <- data[0:train_size];
  test_data <- data[train_size:data_length];
  res <- list("train" = train_data, "test" = test_data);
  return(res);
}

text_data <- split_data(text_column);
train_text <- text_data$train;

# Byte-pair encoding
bpe_model <- bpe(train_text);
bpe_encode <- bpe_encode(bpe_model, x = text, type = "subwords");
head(unlist(bpe_encode), n = 20);

# Create a corpus object and add docvar Target
corpus_docs <- corpus(text_column);
docvars(corpus_docs, field = "Target") <- target_column;

# Create a corpus object of 20 docs and add docvar Target
corpus_docs_cluster <- corpus(text_column[1:20]);
docvars(corpus_docs_cluster, field = "Target") <- target_column[1:20];

# Create a Term-Frequency matrix for both corpus
dfm_docs <- dfm(tokens(corpus_docs));
dfm_docs_cluster <- dfm(tokens(corpus_docs_cluster));

# Clustering process
distance_matrix <- dist(as.matrix(dfm_docs_cluster), method = "euclidean");
clusters <- hclust(distance_matrix, method = "ward.D");
plot(clusters, cex = 0.5, hang = -1); rect.hclust(clusters, k = 2);

# Show most words and lest word of Term-Frequency matrix
topfeatures(dfm_docs); 
topfeatures(dfm_docs, decreasing = FALSE);

# Split into negative, neutral and positives docs and show most used words
negative_docs <- corpus_subset(corpus_docs, Target == "negative"); 
neutral_docs <- corpus_subset(corpus_docs, Target == "neutral"); 
positives_docs <- corpus_subset(corpus_docs, Target == "positive");
negatives_dfm <- dfm(tokens(negative_docs));
neutral_dfm <- dfm(tokens(neutral_docs));
positives_dfm <- dfm(tokens(positives_docs));

topfeatures(negatives_dfm);
topfeatures(neutral_dfm);
topfeatures(positives_dfm);

# Split into train and test subsets
len <- length(target_column)
train_size <- 0.7 * len;
set <- vector();
set[0:train_size] <- TRUE;
set[train_size:len] <- FALSE;
set <- set[0:len]

dfm_train <- dfm_subset(dfm_docs, set);
dfm_test <- dfm_subset(dfm_docs, !set);

# Document classification: Naive Bayes
produce_nb_model <- function(distribution){ 
  naive_bayes_model <- textmodel_nb(dfm_train, docvars(dfm_train)$Target, distribution = distribution);
  prediction <- predict(naive_bayes_model, dfm_test);
  confusion_matrix <- confusionMatrix(prediction, docvars(dfm_test)$Target);
  print(confusion_matrix);
  
  return(confusion_matrix);
} 
confusion_matrix <- produce_nb_model("multinomial");

# Document classification: SVM Model
produce_svm_model <- function(weight){ 
  naive_bayes_model <- textmodel_svm(dfm_train, docvars(dfm_train)$Target, weight = weight);
  prediction <- predict(naive_bayes_model, dfm_test);
  confusion_matrix <- confusionMatrix(prediction, docvars(dfm_test)$Target);
  print(confusion_matrix);
  return(confusion_matrix);
} 
confusion_matrix <- produce_svm_model("uniform");

# Final
spacy_finalize()

sessionInfo()
