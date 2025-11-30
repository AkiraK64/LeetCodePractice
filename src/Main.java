public class Main {
    public static void main(String[] args) {
        var solution = new NeetcodeSolution();
        var dailySolution = new DailyLeetcodeSolution();
        var weeklySolution = new WeeklyLeetcodeSolution();
        int[] nums = new int[]{8,32,31,18,34,20,21,13,1,27,23,22,11,15,30,4,2};
        int p = 148;
        int sum = 0;
        int subSum = 0;
        for(var num : nums) sum += num;
        for(int i=0;i<7;i++){
            subSum += nums[i];
        }
        boolean check = (sum - subSum) % p == 0;
        if(check)
            System.out.println("0, 6");
        for(int i=7;i<nums.length;i++){
            subSum += nums[i];
            subSum -= nums[i-7];
            check = (sum - subSum) % p == 0;
            if(check)
                System.out.println(i-6 + "," + i);
        }
        System.out.println(dailySolution.minSubarray(nums, p));
    }
}