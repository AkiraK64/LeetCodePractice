public class Main {
    /** Practice Dairy
     * Neetcode:
     * Two Pointers - 11 (1/3)
     * Array & Hashing - 103 (1/3)
     * Daily:
     * 2141 - (2/3)
     */
    public static void main(String[] args) {
        var solution = new NeetcodeSolution();
        var dailySolution = new DailyLeetcodeSolution();
        var weeklySolution = new WeeklyLeetcodeSolution();
        int[] nums = new int[]{3,5,10,10};
        int k = 3;
        System.out.println(dailySolution.maxRunTime(k, nums));
    }
}