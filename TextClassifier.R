# init
libs <-c("tm", "plyr", "class")
lapply(libs, require, character.only = TRUE)

# Set options
options(stringsAsFactors = FALSE)

# Set parameters
candidates <- c("romney", "obama")
pathname <- "C:/Users/.../speeches"

# clean text
cleanCorpus <- function(corpus) {
  corpus.tmp <- tm_map(corpus, removePunctuation)
  corpus.tmp <- tm_map(corpus.tmp, stripWhitespace)
  corpus.tmp <- tm_map(corpus.tmp, tolower)
  corpus.tmp <- tm_map(corpus.tmp, removeWords, stopwords("english"))
  return(corpus.tmp)
}

# build TDM
generateTDM <- function(cand, path) {
  s.dir <- sprintf("%s/%s", path, cand)
  s.cor <- Corpus(DirSource(directory = s.dir, encoding = "ANSI"))
  s.cor.cl <- cleanCorpus(s.cor)
  s.tdm <- TermDocumentMatrix(s.cor.cl)
  
  s.tdm <- removeSparseTerms(s.tdm, 0.7)
  result <- list(name = cand, tdm = s.tdm)
}

tdm <- lapply(candidates, generateTDM, path = pathname)

# atatch name
bindCandidateToTDM <- function(tdm) {
  s.mat <- t(data.matrix(tdm[["tdm"]]))
  s.df <- as.data.frame(s.mat, stringAsFactors = FALSE)
  
  s.df <- cbind(s.df, rep(tdm[["name"]], nrow(s.df)))
  colnames(s.df)[ncol(s.df)] <- "targetCandidate"
  return(s.df)
}

candTDM <- lapply(tdm, bindCandidateToTDM)

# stack
tdm.Stack <- do.call(rbind.fill, candTDM)
tdm.stack[is.na[tdm.stack]] <- 0

# hold-out
train.idx <- sample[nrow(tdm.stack), ceiling(nrow(tdm.stack) * 0.7)]
test.idx <-(1:nrow(tdm.stack))[-train.idx]

# model - KNN
model.cand <- tdm.stack[, "targetCandidate"]
tdm.stack.nl <- tdm.stack[, !colnames(tdm.stack) %in% "targetCandidate"]

knn.pred <- knn(tdm.stack.nl[train.idx, ], tdm.stack.nl(test.idx, ), tdm.cand(train.idx))

# accuracy
conf.mat <- table("Predictions" = knn.pred, Actual = tdm.cand[test.idx])
(accuracy <- sum(diag(conf.mat)) / length(text.idx) * 100)
