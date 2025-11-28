import java.util.Comparator;
import java.util.Stack;

/** <i>Stack - 155. MinStack</i> */
class MinStack {
    Stack<Integer> stack;
    Stack<Integer> minStack;

    public MinStack() {
        stack = new Stack<>();
        minStack = new Stack<>();
    }

    public void push(int val) {
        stack.push(val);
        if(minStack.isEmpty()) minStack.push(val);
        else {
            int minValue = minStack.peek();
            if (minValue >= val) minStack.push(val);
        }
    }

    public void pop() {
        if(stack.isEmpty()) return;
        int removedValue = stack.pop();
        int minValue = minStack.peek();
        if (minValue == removedValue) minStack.pop();
    }

    public int top() {
        return stack.isEmpty() ? -1 : stack.peek();
    }

    public int getMin() {
        return minStack.isEmpty() ? -1 : minStack.peek();
    }
}