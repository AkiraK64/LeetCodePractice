public class Main {
    /** Practice Dairy
     * Neetcode:
     * Two Pointers - 11 (1/3)
     * Array & Hashing - 128 (1/3)
     * Daily:
     * 2141 - (2/3)
     * 3625 - (1/2)
     */
    public static void main(String[] args) {
        var solution = new NeetcodeSolution();
        var dailySolution = new DailyLeetcodeSolution();
        var weeklySolution = new WeeklyLeetcodeSolution();
        int[][] points = new int[][]{{71,-89},{-75,-89},{-9,11},{-24,-89},{-51,-89},{-77,-89},{42,11}};
        System.out.println(dailySolution.countTrapezoidsII(points));
    }
}