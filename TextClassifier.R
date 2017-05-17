# Set seed for reproducible results
set.seed(100)

# Packages init
libs <-c("tm", "plyr", "class", "SnowballC") # "tm" - Text mining: Corpus and Document Term Matrix; "class" - KNN model; "SnowballC" - Stemming words
lapply(libs, require, character.only = TRUE)

# Read csv with two columns: text and category
df <- read.table(unz("data/smsspamcollection.zip", "SMSSpamCollection"), 
                   sep="\t", header=F, stringsAsFactors=F, quote="", col.names=c("Category", "Message"))

# Create corpus
docs <- Corpus(VectorSource(df$Message))

# Clean corpus
docs <- tm_map(docs, content_transformer(tolower))
docs <- tm_map(docs, removeNumbers)
docs <- tm_map(docs, removeWords, stopwords("english"))
docs <- tm_map(docs, removePunctuation)
docs <- tm_map(docs, stripWhitespace)
docs <- tm_map(docs, stemDocument, language = "english")

# Create dtm
dtm <- DocumentTermMatrix(docs)
dtm <- removeSparseTerms(dtm, 1-0.001)

# Transform dtm to matrix to data frame - df is easier to work with
mat.df <- as.data.frame(data.matrix(dtm), stringsAsfactors = FALSE)

# Column bind category (known classification)
mat.df <- cbind(mat.df, df$Category)

# Change name of new column to "category"
colnames(mat.df)[ncol(mat.df)] <- "category"

# Split data by rownumber into two equal portions
train <- sample(nrow(mat.df), ceiling(nrow(mat.df) * .70))
test <- (1:nrow(mat.df))[- train]

# Isolate classifier
cl <- mat.df[, "category"]

# Create model data and remove "category"
modeldata <- mat.df[,!colnames(mat.df) %in% "category"]

# Create model: training set, test set, training set classifier
knn.pred <- knn(modeldata[train, ], modeldata[test, ], cl[train])

# Confusion matrix
conf.mat <- table("Predictions" = knn.pred, Actual = cl[test])
conf.mat

# Accuracy
(accuracy <- sum(diag(conf.mat))/length(test) * 100)