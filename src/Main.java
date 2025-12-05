public class Main {
    /** Neetcode:
     * Dairy:
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