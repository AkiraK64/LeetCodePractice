import java.util.ArrayList;
import java.util.List;

/** <i>Arrays & Hashing</i> - 706. Design HashMap */
class MyHashMap {
    List<MyPair> myMap;
    public MyHashMap() {
        myMap = new ArrayList<>();
    }

    public void put(int key, int value) {
        int l = 0;
        int r = myMap.size() - 1;
        while (l <= r){
            int mid = (l + r)/2;
            if(myMap.get(mid).key == key){
                myMap.get(mid).value = value;
                return;
            }
            else if(myMap.get(mid).key < key) r = mid - 1;
            else l = mid + 1;
        }
        myMap.add(l, new MyPair(key, value));
    }

    public int get(int key) {
        int l = 0;
        int r = myMap.size() - 1;
        while (l <= r){
            int mid = (l + r)/2;
            if(myMap.get(mid).key == key) return myMap.get(mid).value;
            else if(myMap.get(mid).key < key) r = mid - 1;
            else l = mid + 1;
        }
        return -1;
    }

    public void remove(int key) {
        int l = 0;
        int r = myMap.size() - 1;
        while (l <= r){
            int mid = (l + r)/2;
            if(myMap.get(mid).key == key){
                myMap.remove(mid);
                return;
            }
            else if(myMap.get(mid).key < key) r = mid - 1;
            else l = mid + 1;
        }
    }
}

class MyPair{
    int key;
    int value;

    MyPair(int key, int value){
        this.key = key;
        this.value = value;
    }
}