SRC 	=	perception.py	\
			my_mip.py	\

NAME	=	my_torch_generator

NAME2	=	my_torch_analyzer

all :
		cp my_torch_generator.py $(NAME)
		cp my_torch_analyzer.py $(NAME2)
		chmod +x $(NAME)
		chmod +x $(NAME2)

clean :
	rm -rf *.pyc
	rm -rf __pycache__/
	rm $(NAME)
	rm $(NAME2)

fclean : clean
	rm -f *~

re : fclean all
