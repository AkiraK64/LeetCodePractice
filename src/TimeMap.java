import java.util.*;

/** <i>Binary Search</i> - 981. Time Based Key-Value Store */
class TimeMap {
    Map<String, TreeMap<Integer, String>> timeMap;
    public TimeMap() {
        timeMap = new HashMap<>();
    }

    public void set(String key, String value, int timestamp) {
        if(timeMap.containsKey(key)){
            var subTree = timeMap.get(key);
            subTree.put(timestamp, value);
        }
        else{
            var subTree = new TreeMap<Integer, String>();
            subTree.put(timestamp, value);
            timeMap.put(key, subTree);
        }
    }

    public String get(String key, int timestamp) {
        if(timeMap.containsKey(key)){
            var subTree = timeMap.get(key);
            var entry = subTree.floorEntry(timestamp);
            return entry == null ? "" : entry.getValue();
        }
        return "";
    }
}
