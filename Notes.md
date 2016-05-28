# TED Talk Transcripts

## Reasons for Interest
        - incredible saturation of various dimensions of culture very
          quickly, both popular and on NPR.
        - impact on education (TEDucation)
        - A medium form: we see repeatedly the gap between the study of
          culture as tweets and the study of culture as novels. There
          is an incredible middle ground and these talks are
          representative of that.
		  
## Features Previously Discussed
        - This initial feature set is part of Sebastian Wernicke’s TED
          talk
          (http://www.ted.com/talks/lies_damned_lies_and_statistics_abou
          t_tedtalks), but his version of events is not alone in trying
          to provide a framework or formula for giving great talks or
          speeches. I also want to see if there are features to be
          gleaned from talks listed below:
            - June Cohen: What Makes a Great TEDTalk
              (https://youtu.be/RVDfWfUSBIM)
            - Nancy Duarte: The secret structure of great talks
              (https://youtu.be/1nYFpuc2Umk)
            - How to TEDx: How to give a great TEDx Talk
              (https://youtu.be/6H67JgLwyF4)
            - Will Stephen has a funny take on all this:
              https://youtu.be/8S0FDjFBj8o
        - Wernicke’s claims
            - 1 week of video, 1 million words, 2 million ratings
            - Goal: the ultimate TED talk and the worst possible (but
              still acceptable)
            - Foci: topics used, style of delivery, and visuals used.
            - Sorted talks in two dimensions: ideas/actions,
              emotional/rational.
            - Top ten words in the most favorite TED talks and the
              least favorite. > Then the topics that TED audiences like
              and those they do not like.
            - Also compared lengths (in time) of the most and least
              favorited talks. (Included comments in this because he
              was able to report that shorter talks got comments like
              “beautiful.”
            - 4-grams for most and least favorite.
            - He has a few more things to consider (e.g., the “TED
              Pad”).
## Possible Features to Consider
        - What is the overall “topical map” of the TED talks and what
          is changing map of the talks: i.e., year to year.
        - Do topics arise in response to external events?
        - What is the most central TED talk once we have a set of
          features?
        - Do the early talks set any kind of example or reveal any kind
          of influence in later talks in terms of topics or word use?
        - Are their consistent phrases? (See Wernicke above.)
        - Do these talks reveal any particular shape? (In terms of
          sentiment or other controlled-vocabulary methods.)
        - Speaking of which, is there a TED vocabulary/lexicon?
        - How do any of these features play out against how often a
          talk is viewed/favorited? (We have another measure of
          sentiment, which means we also have a way to consider how the
          sentiment about a talk compares to sentiment within a talk.)
        - Time 100 vs TED speakers. Which comes first? Who comes across?
            - Time's list is published in April.
        - Sentiment vs Public Mood (Stimson).
        - Some TEDs have tag lines: do talks follow the "assigned
          topic"?
        - When does a TED talk “deliver” its title? (Train topics on
          documents and then hand code titles.)
        - Dynamic Topic Modeling: is there influence? (Probably not but
          if null, interesting?)
        - ted.com/speakers has all the speakers and it has them by some
          sort of category/designation.
            - >>> Grab speaker profiles from TED website, build CSV
              with name, designation, and text. (We will derive gender
              from text.)

## More Notes

The following CFP came across _The Humanist_ mailing list two days ago (May 26). I only quote the top portion below for the notion of *trend detection*:

```
========================================================
CALL FOR PAPERS
20th International Conference on Knowledge Engineering and Knowledge
Management (EKAW 2016)

19-23 November 2016, Bologna, Italy
Abstract submission: July 8, 2016
Paper submission: July 15, 2016

Web site: http://ekaw2016.cs.unibo.it/?q=callforpapers
========================================================


The 20th International Conference on Knowledge Engineering and Knowledge
Management is concerned with the impact of time and space on the
representation of knowledge. Knowledge engineering has mostly been about
creating static, universal representations. Yet the world is rarely static:
everything changes, including the models, and real world systems need to
evolve along with the surrounding world. Also, what makes some
representations valid in some contexts may make them invalid elsewhere
(e.g., jurisdiction for laws).

The special focus of this year's EKAW is "evolving knowledge", which
concerns all aspects of the management and acquisition of knowledge
representations of evolving, contextual, and local models. This includes
change management, trend detection, model evolution, streaming data and
stream reasoning, event processing, time-and space dependent models,
contextual and local knowledge representations, etc.

EKAW 2016 will put a special emphasis on the evolvability and localization
of knowledge and the correct usage of these limits.
```