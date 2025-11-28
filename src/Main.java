import java.util.BitSet;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        var solutionOne = new FirstSolution();
        var dailySolution = new DailySolutionOne();
        int[] nums = new int[]{-5,1,2,-3,4};
        System.out.print(dailySolution.maxSubarraySum(nums, 2));
    }
}