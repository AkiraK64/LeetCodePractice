public class Main {
    /** Neetcode:
     * Dairy:
     */
    public static void main(String[] args) {
        var solution = new Solution();
        var dailySolution = new DailySolution();
        var weeklySolution = new WeeklySolution();
        int[] prices = new int[]{4,7,13};
        int[] strategy = new int[]{-1,-1,0};
        int k = 2;
        System.out.println(dailySolution.maxProfit(prices, strategy, k));
    }
}