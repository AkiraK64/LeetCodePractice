import java.util.Stack;

/** <i>Stack</i> - 901. Online Stock Span */
class StockSpanner {
    Stack<int[]> st;
    public StockSpanner() {
        st = new Stack<>();
    }

    public int next(int price) {
        int[] nextPrice = new int[]{price, 1};
        while (!st.isEmpty()){
            var curPrice = st.peek();
            if(curPrice[0] <= price){
                st.pop();
                nextPrice[1] += curPrice[1];
            }
            else break;
        }
        st.add(nextPrice);
        return nextPrice[1];
    }
}