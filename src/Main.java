public class Main {
    /** Neetcode:
     * Dairy:
     */
    public static void main(String[] args) {
        var solution = new Solution();
        var dailySolution = new DailySolution();
        var weeklySolution = new WeeklySolution();
        var buildings = new int[][]{{1,2},{2,2},{3,2},{2,1},{2,3}};
        System.out.println(dailySolution.countCoveredBuildings(3, buildings));
    }
}