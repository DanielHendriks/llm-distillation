library(tidyverse)
library(ggplot2)
library(reshape2)
library(ggbeeswarm)
library(ggpubr)
library(rstatix)
library(car)
library(broom)
library(qqplotr)
library(npmv)
library(rcompanion)
library(plyr)
library("ggsci")

data <- read.csv("../Data/study.csv")

### data preparation
data <- data %>% 
  mutate(model = as.factor(model),
         gender = as.factor(gender),
         age = as.factor(age),
         country = as.factor(country),
         education = as.factor(education),
         employement = as.factor(employement),
         Quality = rowMeans(select(., c(Plausibility, Understandability, 
                                        Completeness, Satisfaction, Contrastiveness)))) %>% 
  add_column(id = 1:nrow(data), .before = 1) %>% 
  reorder_levels(model, order = c("CF:PosRationales", "MT:PosRationales", "MT+CF:PosRationales", "MT+CF:RevRationales"))

data$model <- revalue(data$model, c("CF:PosRationales" = "CF:Unrevised", 
                                    "MT:PosRationales" = "MT:Unrevised", 
                                    "MT+CF:PosRationales" = "MT+CF:Unrevised", 
                                    "MT+CF:RevRationales" = "MT+CF:Revised"))


count(data$gender)
### graphs
# data %>% 
#   melt(id = 'model', measure.vars = c('Plausibility', 'Understandability',
#                                       'Completeness', 'Satisfaction', 'Contrastiveness')) %>% 
#   ggboxplot(data = ., y = 'value', x = 'variable', fill = 'model', palette = "jco")


# data %>% 
#   melt(id = 'model', measure.vars = c('Plausibility', 'Understandability',
#                                       'Completeness', 'Satisfaction', 'Contrastiveness')) %>% 
#   ggplot(aes(value))+
#   geom_histogram() + 
#   facet_grid(cols = vars(model), rows = vars(variable))+
#   theme_minimal()+
#   scale_fill_brewer(palette="Set1")+
#   theme(
#     panel.grid.major.x = element_blank(),
#     panel.grid.minor.x = element_blank(),
#   )

group.colors <- c("CF:Unrevised" = "#747474", 
                  "MT:Unrevised" = "#479E88", 
                  "MT+CF:Unrevised" ="#6CB9D2", 
                  "MT+CF:Revised" = "#405282")

data %>% 
  melt(id = 'model', measure.vars = c('Plausibility', 'Understandability',
                                      'Completeness', 'Satisfaction', 'Contrastiveness', 'Quality')) %>% 
  ggplot(aes(y = value, x = variable, fill = model))+
  geom_bar(position = position_dodge(width = 0.9), stat = "summary", fun = 'mean', width = 0.7) + 
  geom_errorbar(position = position_dodge(width = 0.9), stat = "summary", fun.data = "mean_se", width = 0.2) +
  coord_cartesian(ylim=c(3.5,4.5)) +  
  theme_minimal()+
  scale_fill_manual(values=group.colors) +
  #scale_fill_brewer(palette="Spectral")+
  #scale_fill_npg()+
  labs(y = 'Mean response value', 
       x = 'Construct', 
       fill = "Training Method : Training Data")+
  theme_pubclean(base_size = 12) 
  #annotate("text", x = -0.5, y = 4.55, label = "A", size = 6, fontface = "bold")


outliers_quality <- data %>%
  group_by(model) %>%
  identify_outliers(Quality)

data_quality <- data %>% 
  anti_join(outliers_quality, by = c("id"))

### univariate normality assumption check
shapio.res <- data %>%
  group_by(model) %>%
  shapiro_test(Plausibility, Understandability, Completeness, Satisfaction, Contrastiveness, Quality) %>%
  arrange(variable)


data %>% 
  melt(id = 'model', measure.vars = c('Plausibility', 'Understandability',
                                      'Completeness', 'Satisfaction', 'Contrastiveness', 'Quality')) %>% 
  ggplot(aes(sample = value))+
  stat_qq_point() + stat_qq_line() + stat_qq_band() + 
  facet_grid(rows = vars(model), cols = vars(variable)) + 
  theme_minimal() +
  scale_fill_brewer(palette="Set1")+
  labs(y = 'Empirical', x = 'Theoretical')+
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
  )


### multivariate normality check
data %>%
  select(Plausibility, Understandability, Completeness, Satisfaction, Contrastiveness) %>%
  mshapiro_test()


### correlation test
corr.res <- data %>% 
  cor_test(Plausibility, Understandability, Completeness, Satisfaction, Contrastiveness)

### Check the homogeneity of covariances assumption
box_m(data[, c("Plausibility", "Understandability", "Completeness", "Satisfaction", "Contrastiveness")], data$model)

### Check the homogneity of variance assumption
levenve.res <- data %>% 
  #gather(key = "variable", value = "value", Plausibility, Understandability, Completeness, Satisfaction, Contrastiveness) #%>%
  melt(id = 'model', measure.vars = c('Plausibility', 'Understandability',
                                      'Completeness', 'Satisfaction', 'Contrastiveness', 'Quality')) %>% 
  group_by(variable) %>%
  levene_test(value ~ model)

### use average as construct
# with data that excludes outliers
model4 <- lm(Quality ~ model + rationale_length, data_quality)
summary(model4) 

qqnorm(resid(model4))
qqline(resid(model4))

par(mfrow=c(2,2))
plot(model4, which=1:4)

hist(model4$residuals, labels = )


# linear model with sociodemographic controls
data_quality$gender <- relevel(data_quality$gender, ref = "male")
data_quality$country <- relevel(data_quality$country, ref = "United Kingdom")

model5 <- lm(Quality ~ model + rationale_length + gender + age + country + education + employement, data_quality)
summary(model5)


### npmv

# par(mfrow=c(3,2))
npmv <- nonpartest(Plausibility|Understandability|Completeness|Satisfaction|Contrastiveness~model, data, permreps=1000, plots = FALSE)


### Kruskal-Wallis Test
res.kruskal <- data %>% 
  melt(id = 'model', measure.vars = c('Plausibility', 'Understandability',
                                 'Completeness', 'Satisfaction', 'Contrastiveness')) %>% 
  group_by(variable) %>% 
  kruskal_test(value ~ model)
res.kruskal

data %>% 
  melt(id = 'model', measure.vars = c('Completeness', 'Contrastiveness')) %>% 
  # filter(variable %in% c("Plausibility", "Completeness", "Contrastiveness")) %>% 
  group_by(variable) %>% 
  dunn_test(value ~ model, p.adjust.method = "bonferroni") %>% 
  View()


vda.completeness <- multiVDA(Completeness ~ model, data = data, digits = 3, statistic = "VDA")

vda.contrastiveness <- multiVDA(Contrastiveness ~ model, data = data, digits = 3, statistic = "VDA")
