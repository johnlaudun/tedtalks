library(ggplot2)
df <- read.csv("../output/tfidf_by_speaker_trimmed.csv")

diff_df <- merge(df[df$speaker_gender == "man",], 
                 df[df$speaker_gender == "woman",], 
                 by=c('subject', 'verb'))
diff_df['diff'] <- diff_df$tfidf.x - diff_df$tfidf.y

diff_df_i <- diff_df[diff_df$subject == "i",]
diff_df_he <- diff_df[diff_df$subject == "he",]
diff_df_she <- diff_df[diff_df$subject == "she",]

i_df <- df[df$subject == "i",]
i_df$verb <- factor(i_df$verb, 
                    levels=diff_df_i$verb[order(diff_df_i$diff)])
i_df <- i_df[!is.na(i_df$verb),]

he_df <- df[df$subject == "he",]
he_df$verb <- factor(he_df$verb, 
                     levels=diff_df_he$verb[order(diff_df_he$diff)])
he_df <- he_df[!is.na(he_df$verb),]

she_df <- df[df$subject == "she",]
she_df$verb <- factor(she_df$verb, 
                      levels=diff_df_she$verb[order(diff_df_she$diff)])
she_df <- she_df[!is.na(she_df$verb),]

ggplot(i_df, aes(x=tfidf, y=verb, color=speaker_gender)) +
  geom_point() + scale_x_log10() + 
  geom_segment(data=diff_df_i, aes(x=tfidf.x, xend=tfidf.y, y=verb, yend=verb), color="black") +
  geom_point() + scale_color_manual(values=c("#4775ba", "#ba4767"))

ggsave("../output/i_subjects_trimmed_new.pdf", width=6, height = 45, units="in")


ggplot(he_df, aes(x=tfidf, y=verb, color=speaker_gender)) +
  geom_point() + scale_x_log10() + 
  geom_segment(data=diff_df_he, aes(x=tfidf.x, xend=tfidf.y, y=verb, yend=verb), color="black") +
  geom_point() + scale_color_manual(values=c("#4775ba", "#ba4767"))

ggsave("../output/he_subjects_trimmed.pdf", width=6, height = 30, units="in")

ggplot(she_df, aes(x=tfidf, y=verb, color=speaker_gender)) +
  geom_point() + scale_x_log10() + 
  geom_segment(data=diff_df_she, aes(x=tfidf.x, xend=tfidf.y, y=verb, yend=verb), color="black") +
  geom_point() + scale_color_manual(values=c("#4775ba", "#ba4767"))

ggsave("../output/she_subjects_trimmed.pdf", width=6, height = 25, units="in")

