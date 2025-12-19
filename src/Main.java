public class Main {
    /** Neetcode:
     * Dairy:
     */
    public static void main(String[] args) {
        var solution = new Solution();
        var dailySolution = new DailySolution();
        var weeklySolution = new WeeklySolution();
        char[][] board = new char[][]{{'A','B','C','E'},{'S','F','C','S'},{'A','D','E','E'}};
        String word = "ABCCDE";
        System.out.println(solution.exist(board, word));
    }
}