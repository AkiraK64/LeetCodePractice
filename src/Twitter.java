import java.util.*;

/** <i>Heap & Priority Queue</i> - 355. Design Twitter */
class TwitterUser{
    int userId;
    List<Integer> followees;
    TreeMap<Integer, Integer> news;

    public TwitterUser(int userId){
        this.userId = userId;
        followees = new ArrayList<>();
        news = new TreeMap<>(Collections.reverseOrder());
    }

    public void post(int timeStamp, int tweetId){
        news.put(timeStamp, tweetId);
        if(news.size() > 10)
            news.remove(news.lastKey());
    }

    public void follow(int followeeId){
        followees.add(followeeId);
    }

    public void unfollow(int followeeId){
        followees.remove((Integer)followeeId);
    }
}

class Twitter {
    Map<Integer, TwitterUser> userMap;
    int timeStamp;
    public Twitter() {
        userMap = new HashMap<>();
        timeStamp = 0;
    }

    public void postTweet(int userId, int tweetId) {
        userMap.computeIfAbsent(userId, k -> new TwitterUser(userId)).post(timeStamp, tweetId);
        timeStamp += 1;
    }

    public List<Integer> getNewsFeed(int userId) {
        var user = userMap.computeIfAbsent(userId, k -> new TwitterUser(userId));
        var pqRes = new TreeMap<Integer, Integer>(user.news);
        TwitterUser followee;
        for(var followeeId : user.followees){
            followee = userMap.get(followeeId);
            pqRes.putAll(followee.news);
            while (pqRes.size() > 10)
                pqRes.remove(pqRes.lastKey());
        }
        return new ArrayList<>(pqRes.values());
    }

    public void follow(int followerId, int followeeId) {
        userMap.computeIfAbsent(followeeId, k -> new TwitterUser(followeeId));
        userMap.computeIfAbsent(followerId, k -> new TwitterUser(followerId)).follow(followeeId);
    }

    public void unfollow(int followerId, int followeeId) {
        userMap.computeIfAbsent(followeeId, k -> new TwitterUser(followeeId));
        userMap.computeIfAbsent(followerId, k -> new TwitterUser(followerId)).unfollow(followeeId);
    }
}