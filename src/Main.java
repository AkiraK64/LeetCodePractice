public class Main {
    /** Neetcode:
     * Dairy: 3578 (1/3)
     */
    public static void main(String[] args) {
        var solution = new Solution();
        var dailySolution = new DailySolution();
        var weeklySolution = new WeeklySolution();
        int[] nums = new int[]{4,5,6,7,0,1,2};
        System.out.println(solution.searchII(nums, 0));
    }
}