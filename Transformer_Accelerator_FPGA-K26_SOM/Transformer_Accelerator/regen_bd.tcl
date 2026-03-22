open_project Transformer_Accelerator.xpr
set bd_file [get_files *.bd]
if {$bd_file ne ""} {
    puts "Resetting output products for $bd_file"
    reset_target all $bd_file
    puts "Generating output products for $bd_file"
    generate_target all $bd_file
    make_wrapper -files $bd_file -top
} else {
    puts "No BD found"
}
close_project
