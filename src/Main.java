public class Main {
    /** Neetcode:
     * Dairy: 3578 (1/3)
     */
    public static void main(String[] args) {
        var solution = new Solution();
        var dailySolution = new DailySolution();
        var weeklySolution = new WeeklySolution();
        int[] t1 = new int[]{1,2,3};
        int[] t2 = new int[]{4,5,6};
        int k = 3;
        System.out.println(weeklySolution.maxPoints(t1, t2, 2));
    }
}