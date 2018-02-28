# Twitter to File


import tweepy
import jsonpickle
from tweepy import OAuthHandler


# =============================================================================
# Gather data from Twitter. Nice source here http://www.dealingdata.net/2016/07/23/PoGo-Series-Tweepy/
# =============================================================================


# Set up Twitter Authentication

consumer_key = 'cdSR83UEppeRT4mCz1Qlkv8pU'
consumer_secret = 'znZWr793z9JEtAcumHAamlsVxmCT7R7zZzVzxfEq3E3qvesPms'
access_token = '966627507318525952-GqzadS5QC31VDwcoQNxyxnfNxoV7cB3'
access_secret = '27MI8JSEdQIpuNCL4E5HTzH3lM9yJKaiiVqqgBJZAFMMv'
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth)

if (not api):
    print ("Problem connecting to API")



#Switching to application authentication
auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)

#Setting up new api wrapper, using authentication only
api = tweepy.API(auth, wait_on_rate_limit=False,wait_on_rate_limit_notify=True) #If put into true, the API wait the 15 minutes and do the new query

#There is a limit in the number of search that we can do
api.rate_limit_status()['resources']['search'] 

# Select US as the place of interest for tweets so we can ahave English ones mainly. We will try the Olympics as a subject.
searchQuery = 'place:96683cc9126741d1 #Olympics'

#Maximum number of tweets we want to collect 
maxTweets = 5000

#The twitter Search API allows up to 100 tweets per query
tweetsPerQry = 1000
tweetCount = 0


#Open a text file to save the tweets to
with open('olympics.json', 'w') as f:

    #Tell the Cursor method that we want to use the Search API (api.search)
    #Also tell Cursor our query, and the maximum number of tweets to return
    for tweet in tweepy.Cursor(api.search,q=searchQuery).items(maxTweets) :         
            
        #Write the JSON format to the text file, and add one to the number of tweets we've collected
        f.write(jsonpickle.encode(tweet._json, unpicklable=False) + '\n')
        tweetCount += 1

    #Display how many tweets we have collected
    print("Downloaded {0} tweets".format(tweetCount))
f.close()


   


