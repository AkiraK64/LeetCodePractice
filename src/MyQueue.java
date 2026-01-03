import java.util.ArrayDeque;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

/** <i>Stack</i> - 232. Impliment Queue Using Stacks */
public class MyQueue {
    List<Integer> queue;
    public MyQueue() {
        queue = new LinkedList<>();
    }

    public void push(int x) {
        queue.add(x);
    }

    public int pop() {
        return queue.removeFirst();
    }

    public int peek() {
        return queue.getFirst();
    }

    public boolean empty() {
        return queue.isEmpty();
    }
}
