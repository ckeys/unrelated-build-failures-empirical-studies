install.packages("Hmisc")
install.packages("DescTools")
install.packages("StepReg")
install.packages("DMwR2")
install.packages("smotefamily")

library("rms")
library("DescTools")
library("StepReg")
library("dplyr")
library("DMwR2")
library("smotefamily")

process_project <- function(project) {
  base_path <- '/Users/andie/PycharmProjects/OtagoPhd/project1/data/r/'
  final_path <- paste0(base_path, project, '_r_analysis_data.csv')

  df <- read.csv(final_path, header = TRUE, check.names = FALSE)
  df <- df[, !colnames(df) %in% c('project_name', 'comment_id', 'issue_id')]
  df <- df[, sapply(df, function(col) length(unique(col[!is.na(col)]))) > 1]

  x <- data.matrix(df)
  cor_matrix <- varclus(x, trans = c("abs"))
  cor_matrix <- round(cor_matrix$sim, 6)

  columns_to_keep <- c("Number of Similar Failures", "CI Latency",
                       "Has Code Patch", "Weekend", "Night Time",
                       "Config Lines Modified")

  high_corr_pairs <- which(cor_matrix > threshold, arr.ind = TRUE)
  row_names <- rownames(cor_matrix)
  col_names <- colnames(cor_matrix)

  exclude_indices <- which(row_names[high_corr_pairs[,1]] != col_names[high_corr_pairs[,2]])
  high_corr_pairs <- high_corr_pairs[exclude_indices, , drop = FALSE]

  features_to_remove <- list()

  for (i in 1:nrow(high_corr_pairs)) {
    pair <- high_corr_pairs[i, ]
    feature1 <- rownames(cor_matrix)[pair[1]]
    feature2 <- colnames(cor_matrix)[pair[2]]
    rm_feature <- ''

    if (feature2 %in% columns_to_keep || feature1 %in% columns_to_keep){
      if (feature1 %in% columns_to_keep && feature2 %in% columns_to_keep){
        rm_feature <- ''
      } else if (feature1 %in% columns_to_keep) {
        rm_feature <- feature2
      } else if (feature2 %in% columns_to_keep) {
        rm_feature <- feature1
      }
    } else{
      rm_feature <- feature1
    }

    print(paste("feature1:", feature1, "feature2:", feature2, "rm_feature:", rm_feature))
    if (rm_feature != '') {
      features_to_remove <- c(features_to_remove, list(rm_feature))
    }
  }

  features_to_remove <- unlist(features_to_remove)
  features_to_remove <- unique(features_to_remove)

  v_filtered <- x[, !(colnames(x) %in% unlist(features_to_remove))]
  v_filtered <- varclus(v_filtered, trans = c("abs"))
  plot(v_filtered)

  remaining_columns <- x[, !(colnames(x) %in% unlist(features_to_remove))]
  remaining_column_names <- colnames(remaining_columns)
  df_filter <- select(df, all_of(remaining_column_names))
  redun_result <- redun(formula = ~., data = df_filter, r2 = 0.9, nk = 0)
  print(redun_result)
  return(list(v_filtered = v_filtered,redun_result = redun_result, remaining_column_names=remaining_column_names, features_to_remove=features_to_remove))
}
project <- 'ambari'
res <- process_project(project)
remaining_column_names<-res$remaining_column_names
quoted_names <- sprintf('"%s"', remaining_column_names)
joined_names <- paste(quoted_names, collapse = ',')
cat(joined_names)