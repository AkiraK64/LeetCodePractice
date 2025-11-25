import java.util.BitSet;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        var solutionOne = new FirstSolution();
        var dailySolution = new DailySolutionOne();
        int[] nums = new int[]{2,3,6,7};
        var res = solutionOne.combinationSum(nums, 7);
    }
}