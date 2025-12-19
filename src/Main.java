public class Main {
    /** Neetcode:
     * Dairy:
     */
    public static void main(String[] args) {
        var solution = new Solution();
        var dailySolution = new DailySolution();
        var weeklySolution = new WeeklySolution();
        int n = 6;
        int firstPerson = 5;
        int[][] meetings = new int[][]{{2,1,1},{0,1,1},{3,2,1},{4,2,2}};
        System.out.println(dailySolution.findAllPeople(n, meetings, firstPerson));
    }
}