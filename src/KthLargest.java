import java.util.ArrayList;
import java.util.PriorityQueue;

/** <i>Heap & Priority - 703. Kth Largest Element in a Stream</i>*/
public class KthLargest {
    PriorityQueue<Integer> minHeap;
    int k;

    public KthLargest(int k, int[] nums) {
        minHeap = new PriorityQueue<>();
        this.k = k;
        for(var e : nums){
            minHeap.add(e);
            if(minHeap.size() > k) minHeap.poll();
        }
    }

    public int add(int val) {
        minHeap.add(val);
        if(minHeap.size() > k) minHeap.poll();
        if(minHeap.isEmpty()) return 0;
        return minHeap.peek();
    }
}
