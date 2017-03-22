module sort(
);

reg [31:0] arr [];
reg [31:0] temp;
int i;
int j;

initial begin
    
    arr = new[255];
    for (i = 0; i < 255; i = i + 1) begin
        arr[i] = $urandom % 100;
        //$display("%d", in[i]);
    end
    
    for (i = 0; i < 255; i = i + 1) begin
        for (j = i; j > 0; j = j - 1) begin
            if (arr[j] < arr[j - 1]) begin
                temp = arr[j];
                arr[j] = arr[j - 1];
                arr[j - 1] = temp;
            end
            else begin
                j = 0;
            end
        end
    end

    for (i = 0; i < 255; i = i + 1) begin
        $display("%d", arr[i]);
    end

end

endmodule
