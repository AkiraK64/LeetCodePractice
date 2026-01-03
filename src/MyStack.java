import java.util.*;

/** <i>Stack</i> - 225. Impliment Stack Using Queues */
class MyStack {
    Deque<Integer> stack;
    public MyStack() {
        stack = new ArrayDeque<>();
    }

    public void push(int x) {
        stack.offer(x);
    }

    public int pop() {
        return stack.pollLast();
    }

    public int top() {
        return stack.peekLast();
    }

    public boolean empty() {
        return stack.isEmpty();
    }
}