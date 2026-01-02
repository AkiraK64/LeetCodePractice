import java.util.ArrayList;
import java.util.List;

/** <i>Arrays & Hashing</i> - 705. Design HashSet */
class MyHashSet {
    List<Integer> mySet;
    public MyHashSet() {
        mySet = new ArrayList<>();
    }

    public void add(int key) {
        int l = 0;
        int r = mySet.size() - 1;
        while (l <= r){
            int mid = (l + r)/2;
            if(mySet.get(mid) == key) return;
            else if(mySet.get(mid) < key) r = mid - 1;
            else l = mid + 1;
        }
        mySet.add(l, key);
    }

    public void remove(int key) {
        int l = 0;
        int r = mySet.size() - 1;
        while (l <= r){
            int mid = (l + r)/2;
            if(mySet.get(mid) == key){
                mySet.remove(mid);
                return;
            }
            else if(mySet.get(mid) < key) r = mid - 1;
            else l = mid + 1;
        }
    }

    public boolean contains(int key) {
        int l = 0;
        int r = mySet.size() - 1;
        while (l <= r){
            int mid = (l + r)/2;
            if(mySet.get(mid) == key) return true;
            else if(mySet.get(mid) < key) r = mid - 1;
            else l = mid + 1;
        }
        return false;
    }
}