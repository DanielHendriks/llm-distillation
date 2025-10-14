library(tidyverse)
library(ggplot2)
library(rstatix)
library(ggpubr)
library(report)
library(emmeans)
library(sciplot)
library(dplyr)
library(ggsignif)
library(plyr)
library("ggsci")
library(ggpattern)

data <- read.csv("../Data/experiment.csv") %>% 
  mutate(target_rationale = as.factor(target_rationale), 
         training_mode = as.factor(training_mode),
         model_size = as.factor(model_size)) %>% 
  mutate(training_data = case_when(
    target_rationale == "few-shot-positive-rationale" ~ "unrevised",
    target_rationale == "few-shot-revision" ~ "revised",
    TRUE ~ as.character(target_rationale)
  ),
  training_mode = case_when(
    training_mode == "task-prefix" ~ "multitask",
    training_mode == "counterfactual-prefix" ~ "counterfactual",
    training_mode == "both" ~ "both",
    TRUE ~ as.character(training_mode)
  ))

data$test.accuracy <- data$test.accuracy * 100
data <- data[!(data$target_rationale == "few-shot-revision" & data$training_mode != "both"),]

data$target_rationale <- revalue(data$target_rationale, 
                                 c("few-shot-positive-rationale" = "Unrevised", 
                                    "few-shot-revision" = "Revised"))

data$training_mode <- revalue(data$training_mode, 
                                 c("multitask" = "MT", 
                                   "counterfactual" = "CF",
                                   "both" = "MT+CF"))

data$model <- paste(data$training_mode, data$target_rationale, sep = ":")
data <- data %>% mutate(model = as.factor(model)) %>% 
  reorder_levels(model, order = c("CF:Unrevised", "MT:Unrevised", "MT+CF:Unrevised", "MT+CF:Revised"))


# plot before outlier removal
group.colors <- c("CF:Unrevised" = "#747474", 
                  "MT:Unrevised" = "#479E88", 
                  "MT+CF:Unrevised" ="#6CB9D2", 
                  "MT+CF:Revised" = "#405282")

data %>% 
  ggplot(aes(x=model, y=test.accuracy, fill = model)) +
  #scale_fill_brewer(palette="Spectral") +
  scale_fill_manual(values=group.colors) +
  #scale_fill_npg()+
  coord_cartesian(ylim=c(50,80)) +
  stat_summary(
    fun = mean, 
    geom = "col", 
    position = "dodge"
  ) +
  stat_summary(
    fun.data = function(y) {
      return(data.frame(
        y = mean(y),
        ymin = mean(y) - se(y),
        ymax = mean(y) + se(y)
      ))
    },
    geom = "errorbar",
    position = position_dodge(width = 0.9),
    width = 0.2
  ) +
  facet_wrap(~model_size)+ 
  theme_pubclean(base_size = 12) + 
  theme(
    axis.text.x = element_text(size = 9)
  ) +
  labs(
    x = "Model size",
    y = "Accuracy",
    fill = "Training Method : Training Data" 
  ) 


### ANOVA
# check for outliers
outliers <- data %>% 
  group_by(model, model_size) %>% 
  identify_outliers(test.accuracy)


# Save original counts before outlier removal
original_counts <- data %>%
  dplyr::group_by(model_size, model) %>%
  dplyr::summarise(original_n = n(), .groups = 'drop')

# Show outliers by group
outlier_counts <- outliers %>%
  dplyr::group_by(model_size, model) %>%
  dplyr::summarise(outliers_n = n(), .groups = 'drop')

print("Outliers identified by group:")
print(outlier_counts)

# Apply the outlier removal (your existing line)
data <- data %>% 
  anti_join(outliers, by = c("model_size", "model", "run"))


# Count remaining samples after outlier removal
final_counts <- data %>%
  dplyr::group_by(model_size, model) %>%
  dplyr::summarise(final_n = n(), .groups = 'drop')

# Create summary of samples dropped
outlier_summary <- original_counts %>%
  dplyr::left_join(outlier_counts, by = c("model_size", "model")) %>%
  dplyr::left_join(final_counts, by = c("model_size", "model")) %>%
  dplyr::mutate(
    outliers_n = ifelse(is.na(outliers_n), 0, outliers_n),
    samples_dropped = original_n - final_n,
    percent_dropped = round((samples_dropped / original_n) * 100, 1)
  ) %>%
  dplyr::arrange(model_size, model)

print("\nOutlier Removal Summary by Group:")
print(outlier_summary)

# Show the specific outlier details
print("\nSpecific outliers removed:")
if(nrow(outliers) > 0) {
  print(outliers %>% dplyr::arrange(model_size, model, run))
} else {
  print("No outliers identified.")
}


# levene test
levene.res <- data %>%
  levene_test(test.accuracy ~ model * model_size) 

# shapiro wilks test
shapiro.res <- data %>%
  group_by(model, model_size) %>%
  shapiro_test(test.accuracy)


# anova 
res.aov <- anova_test(data = data, formula = test.accuracy ~ model * model_size)
anova_res_table <- get_anova_table(res.aov)

# tukey kramer test
tukey.res <- data %>% 
  group_by(model_size) %>% 
  tukey_hsd(test.accuracy ~ model) #%>% View()


data %>% 
  ggplot(aes(x=model, y=test.accuracy, fill = model)) +
  #scale_fill_brewer(palette="Spectral") +
  scale_fill_manual(values=group.colors) +
  #scale_fill_npg()+
  coord_cartesian(ylim=c(50,80)) +
  stat_summary(
    fun = mean, 
    geom = "col", 
    position = "dodge"
  ) +
  stat_summary(
    fun.data = function(y) {
      return(data.frame(
        y = mean(y),
        ymin = mean(y) - se(y),
        ymax = mean(y) + se(y)
      ))
    },
    geom = "errorbar",
    position = position_dodge(width = 0.9),
    width = 0.2
  ) +
  facet_wrap(~model_size)+ 
  theme_pubclean(base_size = 12) + 
  theme(
    axis.text.x = element_text(size = 9)
  ) +
  labs(
    x = "Model size",
    y = "Accuracy",
    fill = "Training Method : Training Data" 
  ) 


# print(data)

# Alternative view: wide format for easier comparison
grouped_averages_wide <- data %>%
  dplyr::group_by(model_size, training_mode, target_rationale) %>%
  dplyr::summarise(
    mean_accuracy = round(mean(test.accuracy, na.rm = TRUE), 2),
    .groups = 'drop'
  ) %>%
  tidyr::pivot_wider(
    names_from = c(training_mode, target_rationale),
    values_from = mean_accuracy,
    names_sep = ":"
  )

print("\nWide format view:")
print(grouped_averages_wide)
