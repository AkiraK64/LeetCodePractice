import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

/** <i>LinkedList</i> - 146. LRU Cache */
class LRUNode{
    int key;
    int value;
    LRUNode next;
    LRUNode prev;
    public LRUNode(int key, int value){
        this.key = key;
        this.value = value;
        next = null;
        prev = null;
    }
}
class LRUCache {
    Map<Integer, LRUNode> map;
    int capacity;
    LRUNode left;
    LRUNode right;

    public LRUCache(int capacity) {
        map = new HashMap<>();
        this.capacity = capacity;
        left = new LRUNode(0, 0);
        right = new LRUNode(0, 0);
        left.next = right;
        right.prev = left;
    }

    void remove(LRUNode node){
        var prev = node.prev;
        var next = node.next;
        prev.next = next;
        next.prev = prev;
    }

    void addLast(LRUNode node){
        var prev = right.prev;
        prev.next = node;
        node.prev = prev;
        node.next = right;
        right.prev = node;
    }

    public int get(int key) {
        if(!map.containsKey(key)) return -1;
        var node = map.get(key);
        remove(node);
        addLast(node);
        return node.value;
    }

    public void put(int key, int value) {
        if(map.containsKey(key)){
            map.get(key).value = value;
            var node = map.get(key);
            remove(node);
            addLast(node);
        }
        else{
            if(map.size() == capacity){
                var first = left.next;
                remove(first);
                map.remove(first.key);
            }
            var node = new LRUNode(key, value);
            map.put(key, node);
            addLast(node);
        }
    }
}