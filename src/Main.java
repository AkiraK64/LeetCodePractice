public class Main {
    /** Neetcode:
     * Dairy:
     */
    public static void main(String[] args) {
        var solution = new Solution();
        var dailySolution = new DailySolution();
        var weeklySolution = new WeeklySolution();
        int[] prices = new int[]{12,16,19,19,8,1,19,13,9};
        int k = 3;
        System.out.println(dailySolution.maximumProfit(prices, k));
    }
}