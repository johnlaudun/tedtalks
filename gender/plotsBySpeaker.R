library(ggplot2)
df <- read.csv("../output/tfidf_by_speaker_trimmed.csv")

# Man speakers

man_df <- df[df$speaker_gender == "man",]
man_diff <- merge(man_df[man_df$subject == "he",], 
                  man_df[man_df$subject == "she",], 
                 by='verb')
man_diff <- merge(man_diff,
                  man_df[man_df$subject == "i",], 
                  by='verb')
# He's are .x, She's are .y, and I's have no dot
man_diff['hs_diff'] <- man_diff$tfidf.x - man_diff$tfidf.y

m_df <- man_diff
m_df$verb <-factor(m_df$verb, 
                   levels = man_diff$verb[order(man_diff$hs_diff)])
m_df <- m_df[!is.na(m_df$verb),]

ggplot(m_df, aes(x=tfidf.y, y=verb, color=subject.y)) +
  geom_point() + 
  scale_x_log10() + 
  geom_segment(data=m_df, aes(x=tfidf.x, xend=tfidf.y, y=verb, yend=verb), color="black") +
  geom_point() + scale_color_manual(values=c("#4775ba", "#ba4767"))


# Woman speakers

woman_df <- df[df$speaker_gender == "woman",]
woman_diff <- merge(woman_df[woman_df$subject == "he",], 
                    woman_df[woman_df$subject == "she",], 
                  by='verb')
woman_diff <- merge(woman_diff,
                    woman_df[woman_df$subject == "i",], 
                  by='verb')
woman_diff['hs_diff'] <- woman_diff$tfidf.x - woman_diff$tfidf.y

wom_df <- woman_diff
wom_df$verb <-factor(wom_df$verb, 
                   levels = woman_diff$verb[order(woman_diff$hs_diff)])
wom_df <- wom_df[!is.na(wom_df$verb),]

ggplot(wom_df, aes(x=tfidf.y, y=verb, color=subject.y)) +
  geom_point() + 
  scale_x_log10() + 
  geom_segment(data=wom_df, aes(x=tfidf.x, xend=tfidf.y, y=verb, yend=verb), color="black") +
  geom_point() + scale_color_manual(values=c("#4775ba", "#ba4767"))
