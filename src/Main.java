public class Main {
    /** Practice Dairy
     * Neetcode:
     * Two Pointers - 11 (2/3)
     * Array & Hashing - 128 (2/3)
     * Stack - 853 (1/2)
     * Daily:
     */
    public static void main(String[] args) {
        var solution = new NeetcodeSolution();
        var dailySolution = new DailyLeetcodeSolution();
        var weeklySolution = new WeeklyLeetcodeSolution();
        int[] piles = new int[]{805306368,805306368,805306368};
        int h = 1000000000;
        System.out.println(solution.minEatingSpeed(piles, h));
    }
}